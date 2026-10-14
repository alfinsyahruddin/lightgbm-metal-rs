#!/usr/bin/env python3
"""ctypes adapter for the identity-checked bundled C++ reference; no Python LightGBM package."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

from build_reference import ROOT, WORK, source_identity


class Reference:
    def __init__(self):
        pin = source_identity()
        provenance = json.loads((WORK / "provenance.json").read_text())
        if provenance["source_tree"] != pin["source_tree"]:
            raise RuntimeError("reference build source mismatch")
        path = ROOT / provenance["library"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["library_sha256"]:
            raise RuntimeError("reference library hash mismatch")
        self.provenance = provenance
        self.lib = C.CDLL(str(path))
        signatures = {
            "LGBM_GetLastError": (C.c_char_p, []),
            "LGBM_DatasetCreateFromMat": (C.c_int, [C.c_void_p, C.c_int, C.c_int32, C.c_int32, C.c_int, C.c_char_p, C.c_void_p, C.POINTER(C.c_void_p)]),
            "LGBM_DatasetSetField": (C.c_int, [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_int]),
            "LGBM_BoosterCreate": (C.c_int, [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]),
            "LGBM_BoosterUpdateOneIter": (C.c_int, [C.c_void_p, C.POINTER(C.c_int)]),
            "LGBM_BoosterPredictForMat": (C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_int32, C.c_int32, C.c_int, C.c_int, C.c_int, C.c_int, C.c_char_p, C.POINTER(C.c_int64), C.POINTER(C.c_double)]),
            "LGBM_BoosterDumpModel": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int64, C.POINTER(C.c_int64), C.c_void_p]),
            "LGBM_BoosterFree": (C.c_int, [C.c_void_p]),
            "LGBM_DatasetFree": (C.c_int, [C.c_void_p]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, args

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.LGBM_GetLastError().decode())

    def train(self, features, labels, cols, config, predict_features=None, dump=True):
        params = {
            "objective": "regression" if config.get("objective", "regression_l2") == "regression_l2" else "binary",
            "learning_rate": config.get("learning_rate", 0.1), "num_leaves": config.get("num_leaves", 31),
            "max_depth": config.get("max_depth") or -1, "max_bin": config.get("max_bin", 255),
            "min_data_in_leaf": config.get("min_data_in_leaf", 20),
            "min_sum_hessian_in_leaf": config.get("min_sum_hessian_in_leaf", 0.001),
            "min_gain_to_split": config.get("min_gain_to_split", 0),
            "lambda_l1": config.get("lambda_l1", 0), "lambda_l2": config.get("lambda_l2", 0),
            "bin_construct_sample_cnt": len(labels), "min_data_in_bin": 3,
            "feature_pre_filter": "false", "enable_bundle": "false", "enable_sparse": "false",
            "use_missing": "true", "zero_as_missing": "false", "device_type": "cpu",
            "deterministic": "true", "force_col_wise": "true", "num_threads": 1,
            "bagging_fraction": 1, "bagging_freq": 0, "feature_fraction": 1,
            "use_quantized_grad": "false", "sigmoid": 1, "verbosity": -1,
        }
        encoded = " ".join(f"{k}={v}" for k, v in params.items()).encode()
        matrix = (C.c_double * len(features))(*(float("nan") if x is None else x for x in features))
        target = (C.c_float * len(labels))(*labels)
        dataset, booster = C.c_void_p(), C.c_void_p()
        started = time.perf_counter()
        try:
            self.check(self.lib.LGBM_DatasetCreateFromMat(matrix, 1, len(labels), cols, 1, encoded, None, C.byref(dataset)))
            self.check(self.lib.LGBM_DatasetSetField(dataset, b"label", target, len(labels), 0))
            prepared = time.perf_counter()
            self.check(self.lib.LGBM_BoosterCreate(dataset, encoded, C.byref(booster)))
            initialized = time.perf_counter()
            for _ in range(config.get("iterations", 100)):
                finished = C.c_int()
                self.check(self.lib.LGBM_BoosterUpdateOneIter(booster, C.byref(finished)))
                if finished.value:
                    break
            fitted = time.perf_counter()
            pred_values = features if predict_features is None else predict_features
            pred_matrix = (C.c_double * len(pred_values))(*(float("nan") if x is None else x for x in pred_values))
            rows = len(pred_values) // cols
            output = (C.c_double * rows)()
            length = C.c_int64()
            self.check(self.lib.LGBM_BoosterPredictForMat(booster, pred_matrix, 1, rows, cols, 1, 1, 0, -1, b"num_threads=1", C.byref(length), output))
            model = None
            if dump:
                size = C.c_int64()
                buf = C.create_string_buffer(1024 * 1024)
                self.check(self.lib.LGBM_BoosterDumpModel(booster, 0, -1, 0, len(buf), C.byref(size), buf))
                if size.value > len(buf):
                    buf = C.create_string_buffer(size.value)
                    self.check(self.lib.LGBM_BoosterDumpModel(booster, 0, -1, 0, len(buf), C.byref(size), buf))
                model = json.loads(buf.value)
            return {"raw_predictions": list(output), "model_dump": model, "parameters": params,
                    "timings": {"preprocessing": prepared - started, "setup": initialized - prepared,
                                "training": fitted - initialized, "end_to_end": fitted - started}}
        finally:
            if booster.value:
                self.check(self.lib.LGBM_BoosterFree(booster))
            if dataset.value:
                self.check(self.lib.LGBM_DatasetFree(dataset))

"""
Constants for the upstream test discovery system.

Test file list, error categories, defaults, and K8s resource names.
"""

# ---------------------------------------------------------------------------
# All 63 PyTorch upstream test files that use instantiate_device_type_tests()
# ---------------------------------------------------------------------------

UPSTREAM_TEST_FILES = [
    "test_binary_ufuncs.py",
    "test_complex.py",
    "test_dlpack.py",
    "test_foreach.py",
    "test_indexing.py",
    "test_linalg.py",
    "test_masked.py",
    "test_namedtensor.py",
    "test_nn.py",
    "test_ops.py",
    "test_ops_fwd_gradients.py",
    "test_ops_gradients.py",
    "test_ops_jit.py",
    "test_proxy_tensor.py",
    "test_reductions.py",
    "test_scatter_gather_ops.py",
    "test_shape_ops.py",
    "test_sort_and_select.py",
    "test_sparse.py",
    "test_sparse_csr.py",
    "test_spectral_ops.py",
    "test_tensor_creation_ops.py",
    "test_torch.py",
    "test_type_hints.py",
    "test_type_promotion.py",
    "test_unary_ufuncs.py",
    "test_view_ops.py",
    "test_vmap.py",
    "test_autograd.py",
    "test_comparison_utils.py",
    "test_decomp.py",
    "test_expanded_weights.py",
    "test_fake_tensor.py",
    "test_functional_autograd_benchmark.py",
    "test_fx_experimental.py",
    "test_fx_passes.py",
    "test_jit.py",
    "test_jit_fuser_te.py",
    "test_jit_llga_fuser.py",
    "test_linalg_grad.py",
    "test_meta.py",
    "test_mkldnn.py",
    "test_modules.py",
    "test_nestedtensor.py",
    "test_overrides.py",
    "test_prims.py",
    "test_quantization.py",
    "test_schema_check.py",
    "test_sparse_semi_structured.py",
    "test_stateless.py",
    "test_subclass.py",
    "test_testing.py",
    "test_transformers.py",
    "test_utils.py",
    "test_xla_sharding.py",
    "functorch/test_aotdispatch.py",
    "functorch/test_control_flow.py",
    "functorch/test_eager_transforms.py",
    "functorch/test_ops.py",
    "functorch/test_vmap.py",
    "inductor/test_decomp.py",
    "inductor/test_torchinductor.py",
    "inductor/test_torchinductor_opinfo.py",
]

# ---------------------------------------------------------------------------
# Error categories for classifying test failures
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = {
    "segfault": {
        "patterns": ["SIGSEGV", "Segmentation fault", "signal 11", "exit code 139"],
        "description": "Segmentation fault / crash",
    },
    "oom": {
        "patterns": ["OutOfMemoryError", "CUDA out of memory", "exit code 137", "OOMKilled"],
        "description": "Out of memory",
    },
    "not_implemented": {
        "patterns": [
            "NotImplementedError",
            "not implemented",
            "aten::.*not found",
            "Could not run.*with arguments from the.*backend",
        ],
        "description": "Op not implemented for Spyre backend",
    },
    "dtype_mismatch": {
        "patterns": [
            "RuntimeError.*dtype",
            "expected.*dtype",
            "Unsupported dtype",
        ],
        "description": "Dtype not supported or mismatch",
    },
    "shape_mismatch": {
        "patterns": [
            "shape.*mismatch",
            "RuntimeError.*size",
            "expected.*dimensions",
        ],
        "description": "Tensor shape or dimension error",
    },
    "precision": {
        "patterns": [
            "AssertionError.*not close",
            "Tensor-likes are not close",
            "values are not close",
            "atol",
            "rtol",
        ],
        "description": "Numerical precision mismatch",
    },
    "timeout": {
        "patterns": ["TimeoutError", "timed out", "timeout"],
        "description": "Test exceeded time limit",
    },
    "import_error": {
        "patterns": ["ImportError", "ModuleNotFoundError", "No module named"],
        "description": "Missing module or import failure",
    },
    "skip": {
        "patterns": ["Skipped", "unittest.skip", "pytest.skip"],
        "description": "Test was skipped",
    },
    "unknown": {
        "patterns": [],
        "description": "Uncategorized failure",
    },
}

# ---------------------------------------------------------------------------
# K8s / cluster defaults
# ---------------------------------------------------------------------------

DEFAULT_NAMESPACE = "torch-spyre-cicd"
DEFAULT_PVC = "my-dev-work-pvc"
DEFAULT_IMAGE = "us.icr.io/wxpe-cicd-internal/amd64/torch-aiu-runtime-dev:latest"
DEFAULT_IMAGE_PULL_SECRET = "wxpe-cicd-vllm-iccr-internal"
DEFAULT_SCHEDULER = "spyre-scheduler"
DEFAULT_SERVICE_ACCOUNT = "default"

DEFAULT_MAX_PARALLELISM = 40
DEFAULT_MIN_PARALLELISM = 1
DEFAULT_RESERVE_CARDS = 5
DEFAULT_PER_POD_TIMEOUT = 14400  # 4 hours
DEFAULT_PER_TEST_TIMEOUT = 300  # 5 minutes
DEFAULT_POLL_INTERVAL = 30  # seconds
DEFAULT_HEARTBEAT_INTERVAL = 60  # seconds
DEFAULT_HEARTBEAT_STALE_THRESHOLD = 900  # 15 minutes
DEFAULT_BACKOFF_LIMIT = 130  # ~2x retries per index

DEFAULT_WORKER_MEMORY_REQUEST = "8Gi"
DEFAULT_WORKER_MEMORY_LIMIT = "32Gi"
HIGH_MEMORY_LIMIT = "64Gi"

# Files known to require extra memory
HIGH_MEMORY_FILES = {
    "test_torch.py",
    "test_nn.py",
    "test_ops.py",
    "test_autograd.py",
    "test_jit.py",
}

DEFAULT_TORCH_SPYRE_REPO = "https://github.com/torch-spyre/torch-spyre.git"
DEFAULT_TORCH_SPYRE_BRANCH = "main"
DEFAULT_PYTORCH_REPO = "https://github.com/pytorch/pytorch.git"
DEFAULT_PYTORCH_BRANCH = "main"

# PVC mount paths
PVC_MOUNT_PATH = "/mnt/devwork"
DISCOVERY_BASE_DIR = "discovery"

# Spyre device env vars
SPYRE_ENV_VARS = {
    "FLEX_COMPUTE": "SENTIENT",
    "FLEX_DEVICE": "PF",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_TESTING_DEVICE_ONLY_FOR": "privateuse1",
    "TORCH_TEST_DEVICES": "privateuse1",
}

# Orchestrator phase names (for state tracking)
PHASES = [
    "init",
    "generate_artifacts",
    "create_job",
    "monitor_job",
    "aggregate",
    "generate_config",
    "diff",
    "finalize",
]

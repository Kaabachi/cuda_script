# GPU Testing Script

This script allows you to check the availability of CUDA, the number of available CUDA devices, and test your GPU with simple or stress tests. It can be run directly or within a Docker container.

## Usage

### Running Directly

1. Install the dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the script with desired options:

    ```bash
    python gpu_script.py [options]
    ```

## Options

- `--check_cuda`: Check if CUDA is available.
- `--check_devices`: Check the number of available CUDA devices.
- `--print_gpu_info`: Print information about available GPU(s).
- `--test [simple|stress-test]`: Test the GPU with a simple or stress test.
- `--tensor_size`: (For stress test only) Size of tensors used in the stress test.

## Examples

- Check if CUDA is available:

    ```bash
    python gpu_script.py --check_cuda 1
    ```

- Check the number of available CUDA devices:

    ```bash
    python gpu_script.py --check_devices 1
    ```

- Print information about available GPU(s):

    ```bash
    python gpu_script.py --print_gpu_info 1
    ```

- Run a simple test:

    ```bash
    python gpu_script.py --test simple
    ```

- Run a stress test with custom tensor size (default size is 10000):

    ```bash
    python gpu_script.py --test stress-test --tensor_size 5000
    ```



import argparse
import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

def check_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print("Number of available CUDA devices:", num_devices)
    else:
        print("CUDA is not available.")

def print_gpu_info():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            device = torch.device("cuda:{}".format(i))
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
            print(torch.cuda.get_device_properties(i))
    else:
        print("CUDA is not available.")

def simple_test():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.rand(3, 3).to(device)
        print("Simple test successful on CUDA device:", tensor.device)
    else:
        print("CUDA is not available for simple test.")

def stress_test(tensor_size):
    if torch.cuda.is_available():
        # Perform some heavy computation to stress the GPU
        device = torch.device("cuda")
        tensor = torch.rand(tensor_size, tensor_size).to(device)
        result = tensor.matmul(tensor)
        print("Stress test successful on CUDA device:", result.device)
    else:
        print("CUDA is not available for stress test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Testing Script")
    parser.add_argument("--check_cuda", type=int, default=0, help="Check if CUDA is available")
    parser.add_argument("--check_devices", type=int, default=0, help="Check number of available CUDA devices")
    parser.add_argument("--print_gpu_info", type=int, default=0, help="Print information about available GPU(s)")
    parser.add_argument("--test", choices=["simple", "stress-test"], help="Test the GPU with simple or stress test")
    parser.add_argument("--tensor_size", type=int, default=10000, help="Size of tensors for stress test")
    args = parser.parse_args()

    if args.check_cuda:
        check_cuda()

    if args.check_devices:
        check_devices()

    if args.print_gpu_info:
        print_gpu_info()

    if args.test == "simple":
        simple_test()

    if args.test == "stress-test":
        stress_test(args.tensor_size)

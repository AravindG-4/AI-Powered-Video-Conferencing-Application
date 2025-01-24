import time
import requests
from concurrent.futures import ThreadPoolExecutor

# Configuration
FASTAPI_URL = "http://localhost:8000/detect-emotions/"
TORCHSERVE_URL = "http://localhost:8080/predictions/emotion_model"
TEST_IMAGE = "surprised.jpg"  # Path to the test image
BATCH_SIZE = 16  # Batch size for testing
NUM_REQUESTS = 100  # Number of requests for scalability test

def load_image_as_bytes(image_path):
    """Load an image and encode it as bytes for HTTP POST."""
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        return None

# Load the test image
test_image_bytes = load_image_as_bytes(TEST_IMAGE)
if not test_image_bytes:
    raise FileNotFoundError("Test image could not be loaded. Please check the image path.")

def benchmark_latency(url, test_image_bytes):
    """Measure the latency of a single request."""
    start_time = time.time()
    try:
        response = requests.post(url, files={"files": ("image.jpg", test_image_bytes)})
        latency = time.time() - start_time
        response.raise_for_status()
        return latency, response.json()
    except requests.RequestException as e:
        return None, {"error": str(e)}

def benchmark_batch_inference(url, test_image_bytes, batch_size):
    """Measure the latency of batch inference."""
    files = [("files", ("image.jpg", test_image_bytes)) for _ in range(batch_size)]
    start_time = time.time()
    try:
        response = requests.post(url, files=files)
        latency = time.time() - start_time
        response.raise_for_status()
        return latency, response.json()
    except requests.RequestException as e:
        return None, {"error": str(e)}

def benchmark_scalability(url, test_image_bytes, num_requests):
    """Test scalability with multiple concurrent requests."""
    def send_request(request_id):
        start_time = time.time()
        try:
            response = requests.post(url, files={"files": ("image.jpg", test_image_bytes)})
            latency = time.time() - start_time
            response.raise_for_status()
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "latency": latency,
                "response": response.json(),
            }
        except requests.RequestException as e:
            return {
                "request_id": request_id,
                "status_code": 503,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(send_request, range(num_requests)))
    total_time = time.time() - start_time
    return total_time, results

def display_results(endpoint, metric_name, metric_value, results=None):
    """Utility to display benchmark results."""
    print(f"\n[{endpoint}] {metric_name}: {metric_value:.4f}s")
    if results:
        print(f"Sample Response: {results}")

# Run benchmarks for FastAPI and TorchServe
def run_benchmarks():
    print("Starting benchmarks...\n")

    # FastAPI Benchmarks
    print("Benchmarking FastAPI...")
    latency, response = benchmark_latency(FASTAPI_URL, test_image_bytes)
    if latency:
        display_results("FastAPI", "Latency", latency, response)

    batch_latency, batch_response = benchmark_batch_inference(FASTAPI_URL, test_image_bytes, BATCH_SIZE)
    if batch_latency:
        display_results("FastAPI", f"Batch Latency (Batch Size {BATCH_SIZE})", batch_latency, batch_response)

    scalability_time, scalability_results = benchmark_scalability(FASTAPI_URL, test_image_bytes, NUM_REQUESTS)
    display_results("FastAPI", f"Scalability ({NUM_REQUESTS} Requests)", scalability_time)
    for result in scalability_results[:5]:  # Display detailed output for first 5 requests
        print(f"Request {result['request_id']} - Latency: {result['latency']:.4f}s, Status: {result['status_code']}, Response: {result.get('response', result.get('error'))}")

    # TorchServe Benchmarks
    print("\nBenchmarking TorchServe...")
    latency, response = benchmark_latency(TORCHSERVE_URL, test_image_bytes)
    if latency:
        display_results("TorchServe", "Latency", latency, response)

    batch_latency, batch_response = benchmark_batch_inference(TORCHSERVE_URL, test_image_bytes, BATCH_SIZE)
    if batch_latency:
        display_results("TorchServe", f"Batch Latency (Batch Size {BATCH_SIZE})", batch_latency, batch_response)

    scalability_time, scalability_results = benchmark_scalability(TORCHSERVE_URL, test_image_bytes, NUM_REQUESTS)
    display_results("TorchServe", f"Scalability ({NUM_REQUESTS} Requests)", scalability_time)
    for result in scalability_results[:5]:  # Display detailed output for first 5 requests
        print(f"Request {result['request_id']} - Latency: {result['latency']:.4f}s, Status: {result['status_code']}, Response: {result.get('response', result.get('error'))}")

# Run the benchmarks
if __name__ == "__main__":
    run_benchmarks()

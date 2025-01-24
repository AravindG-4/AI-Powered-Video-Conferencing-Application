import asyncio
import httpx
import time
from statistics import mean, median, stdev

# API Endpoint for TorchServe
API_ENDPOINT = "http://127.0.0.1:8080/predictions/emotion_model"

# Load the test image
def load_image(image_path):
    """Load the test image into memory."""
    with open(image_path, "rb") as f:
        return f.read()

# Send a single request
async def send_request(client, image_data):
    """Send a single request to the TorchServe API."""
    files = {"data": ("test_image.jpg", image_data, "image/jpeg")}
    try:
        start_time = time.perf_counter()
        response = await client.post(API_ENDPOINT, files=files, timeout=30)  
        elapsed_time = time.perf_counter() - start_time

        if response.status_code == 200:
            print(f"Request Success: {response.json()} | Time: {elapsed_time:.4f}s")
            return {"status": "success", "latency": elapsed_time, "response": response.json()}
        else:
            print(f"Request Failed: {response.status_code} | Time: {elapsed_time:.4f}s")
            return {"status": "failed", "latency": elapsed_time, "code": response.status_code}
    except Exception as e:
        print(f"Request Error: {e}")
        return {"status": "error", "latency": None, "error": str(e)}

# Simulate requests with concurrency and collect results
async def simulate_requests(image_data, total_requests=10000, concurrent_requests=100):
    """Simulate multiple concurrent requests to the API."""
    async with httpx.AsyncClient() as client:
        tasks = []
        results = []

        for i in range(total_requests):
            tasks.append(send_request(client, image_data))

            # Process in batches to avoid overwhelming the system
            if len(tasks) >= concurrent_requests:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []

        # Process remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    return results

# Analyze the results
def analyze_results(results):
    """Analyze the performance metrics from the results."""
    success_latencies = [r["latency"] for r in results if r["status"] == "success"]
    failed_requests = [r for r in results if r["status"] == "failed"]
    error_requests = [r for r in results if r["status"] == "error"]

    if success_latencies:
        avg_latency = mean(success_latencies)
        max_latency = max(success_latencies)
        min_latency = min(success_latencies)
        median_latency = median(success_latencies)
        std_dev_latency = stdev(success_latencies)
    else:
        avg_latency = max_latency = min_latency = median_latency = std_dev_latency = 0

    return {
        "total_requests": len(results),
        "successful_requests": len(success_latencies),
        "failed_requests": len(failed_requests),
        "error_requests": len(error_requests),
        "avg_latency": avg_latency,
        "max_latency": max_latency,
        "min_latency": min_latency,
        "median_latency": median_latency,
        "std_dev_latency": std_dev_latency,
        "success_rate": len(success_latencies) / len(results) * 100 if results else 0,
    }

# Main function
def main():
    image_path = "surprised.jpg"  # Path to the test image
    image_data = load_image(image_path)  # Load the image

    print(f"Starting efficiency test for {API_ENDPOINT} using image {image_path}...")
    start_time = time.time()

    # Run the simulation with defined requests
    total_requests = 10000
    concurrent_requests = 100
    results = asyncio.run(simulate_requests(image_data, total_requests=total_requests, concurrent_requests=concurrent_requests))

    # Analyze results
    stats = analyze_results(results)

    # Print summary
    total_time = time.time() - start_time
    print("\n=== Test Summary ===")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful Requests: {stats['successful_requests']}")
    print(f"Failed Requests: {stats['failed_requests']}")
    print(f"Error Requests: {stats['error_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    print(f"Average Latency: {stats['avg_latency']:.4f} seconds")
    print(f"Maximum Latency: {stats['max_latency']:.4f} seconds")
    print(f"Minimum Latency: {stats['min_latency']:.4f} seconds")
    print(f"Median Latency: {stats['median_latency']:.4f} seconds")
    print(f"Latency Std. Dev.: {stats['std_dev_latency']:.4f} seconds")
    print(f"Total Time Taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

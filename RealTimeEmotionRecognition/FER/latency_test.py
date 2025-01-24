import asyncio
import httpx
import time

# API Endpoint for the new FastAPI application
API_ENDPOINT = "http://127.0.0.1:8000/detect-emotions/"

# Load the test image
def load_image(image_path):
    """Load the test image into memory."""
    with open(image_path, "rb") as f:
        return f.read()

# Send a single request
async def send_request(client, image_data):
    """Send a single request to the FastAPI detect-emotions API."""
    files = {"files": ("image.jpg", image_data, "image/jpeg")}
    try:
        start_time = time.perf_counter()
        response = await client.post(API_ENDPOINT, files=files)
        elapsed_time = time.perf_counter() - start_time

        if response.status_code == 200:
            print(f"Success: {response.json()} | Time: {elapsed_time:.4f}s")
        else:
            print(f"Failed: {response.status_code} | Time: {elapsed_time:.4f}s")
        return elapsed_time
    except Exception as e:
        print(f"Error: {e}")
        return None

# Simulate multiple requests
async def simulate_requests(image_data, total_requests=10000, concurrent_requests=100):
    """Simulate multiple concurrent requests to the API."""
    async with httpx.AsyncClient() as client:
        tasks = []
        latencies = []

        for i in range(total_requests):
            tasks.append(send_request(client, image_data))

            # Process in batches to avoid overwhelming the system
            if len(tasks) >= concurrent_requests:
                results = await asyncio.gather(*tasks)
                latencies.extend(filter(None, results))  # Collect successful latencies
                tasks = []  # Reset tasks

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            latencies.extend(filter(None, results))

    return latencies

# Main function
def main():
    image_path = "surprised.jpg"  # Path to the test image
    image_data = load_image(image_path)  # Load the image

    print(f"Starting latency test with {API_ENDPOINT} using image {image_path}...")
    start_time = time.time()

    # Run the simulation
    latencies = asyncio.run(simulate_requests(image_data, total_requests=10000, concurrent_requests=100))

    # Calculate statistics
    total_time = time.time() - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    print("\n=== Test Summary ===")
    print(f"Total Requests: {len(latencies)}")
    print(f"Total Time Taken: {total_time:.2f} seconds")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"Maximum Latency: {max_latency:.4f} seconds")
    print(f"Minimum Latency: {min_latency:.4f} seconds")

if __name__ == "__main__":
    main()


#torchserve --start --model-store model_store --models emotion_model=emotion_model.mar
import numpy as np


def kmeans(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1), axis=-1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    print(centroids)

    # Calculate the sum of distances between data points and their corresponding centroids
    distances = np.linalg.norm(data - centroids[labels], axis=1)
    print(distances)
    total_distance = np.sum(distances)
    return total_distance


data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [8, 10, 12], [14, 12, 10]])

print(kmeans(data, 2))

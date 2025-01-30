from MNIST import addition, multiplication
import numpy as np
train_addition_dataset = addition(n=2, dataset="train", seed=42)  # Addition of 2-digit numbers
# test_multiplication_dataset = multiplication(n=3, dataset="test", seed=100) # Multiplication of 3-digit numbers

for i in range(1000):
    l1, l2, label, digits = train_addition_dataset[i] # Get the first example

    # print(f"First number images (l1): {l1.shape}")  # Shape of the image tensor for the first number
    # print(f"Second number images (l2): {l2.shape}")  # Shape of the image tensor for the second number
    print(f"Label (result): {label}")  # The actual numerical result of the addition
    print(f"Digits: {digits}")  # The digits that make up the numbers
    print(l1)
    # display the images
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.imshow(   # Display the first number
        np.concatenate([l1[0], l1[1]], axis=1), cmap="gray"
    )
    plt.show()
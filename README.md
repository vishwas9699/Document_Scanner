# Document Scanner

## Document Scanner with OpenCV and Python

I used this Project to run this in real time to scan document and save these images by pressing just "S" button.

The scanner takes a poorly scanned image, finds the corners of the document, applies the perspective transformation to get a top-down view of the document, sharpens the image, and applies an adaptive color threshold to clean up the image.

## Environment Used
* Python
* OpenCV

## Building a document scanner with OpenCV can be accomplished in just three simple steps:

Step 1: Caputre the image
Step 1: Detect edges.
Step 2: Use the edges in the image to find the contour (outline) representing the piece of paper being scanned.
Step 3: Apply a perspective transform to obtain the top-down view of the document.

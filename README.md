
# Lossless, a Go-Based ML Library
## Introduction and Overview
Hi, my name is Egan Bosch, and I am currently a fourth-year computational mathematics major at UCLA. I have fallen in love with machine learning as of late, and also have a bit of an obsession with doing things myself from scratch. So, as a way to teach myself Go, as well as put some of the things I'd been learning into practice, I created my first Go ML library. Now, it worked, but it was slow, as I had created a kind of complicated "expression" data structure to automatically calculate gradients for me. However, this presented two problems:
1. No matter what I did to try and increase the performance of my data structure it simply did not scale well to very large or very deep networks.
2. My network could only train on one input at a time as a consequence to how things were set up.

Not ideal. So, a few months later, feeling inspired and after a sufficiently long pondering session I essentially rederived how to do backpropagation for basic MLP's (which is not insane, I'm aware) but the simplicity of doing just a few matrix multiplications got me excited enough to create Lossless, which has now evolved somewhat. It is still young, so expect updates in the future, and let me know if you wish to collaborate!
## Getting Started
1. Download the latest version of Go [here](https://go.dev/dl/).
2. Create your project directory, and run `go mod init [your project name]`
3. Run `go get github.com/EganBoschCodes/lossless`
4. Get to creating!
## Example Projects
**[Basic Perceptron:](https://github.com/EganBoschCodes/Lossless-MLP-Example)** This example walks you through the basics of getting started with Lossless, and training/saving your first network.

**[Convolutional Neural Network:](https://github.com/EganBoschCodes/Lossless-MNIST-Example)** This example will walk you through how to import tabular data with Lossless' dataframes, save them into a usable format, and then train a convolutional neural network on the MNIST handwritten digits dataset.

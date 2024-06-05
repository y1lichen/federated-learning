## Dependency

Flower IOS SDK must be install and added to project. Flwr package can be downloaded [here](https://github.com/adap/flower/tree/main/src/swift/flwr).
## Adding further Scenarios

If you want to add more scenarios beyond MNIST, do the following:

- Open the _scenarios.ipynb_ notebook and adapt it to your needs based on the existing structure
- Open Xcode and add the dataset(s) and model to the sources of your project
- Add the dataset(s) to _Copy Bundle Resources_ in the Build Phases settings of the project
- Navigate to the _Constants.swift_ file and add your scenario so that it fits into the given structure

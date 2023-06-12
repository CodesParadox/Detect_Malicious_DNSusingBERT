
# DNS Classification using BERT Models

This project focuses on DNS classification using BERT models. It explores three different approaches to classify DNS requests based on their BERT embeddings. 
The models include an ANN model, a Manhattan distance-based model, and an Euclidean distance-based model. 
The BERT embeddings are generated from the "qname" field of the DNS requests.

## Project Structure

The project consists of the following files and directories:

- `data/`: Contains the dataset used for training and testing the models. - the data is classified so you can have it only if you got a permission
- `models/`: Contains the trained models.
-  `old_code_files/`: Contains all the old code that we use - if you dont need it avoid it.
- `Features_extraction.py`: Contains utility functions and scripts used in the project, that manly focus on getting the feature.
- `Load_and_Save.py`: Contains utility functions and scripts used in the project, that manly focus on loading and saving the differnt files the feature.
- `Prediction_and_Evaluation.py`: this class predict and evaluate the model, it print out or and save the confusion matrix with added calculated fields.
- `TrainModels.py`: this class train the differnt models and save them.
- `test.py`: The test script to run and evaluate the models. 
- `requirements.txt`: Lists the required dependencies for the project.

## Installation

To run the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/dns-classification.git
cd dns-classification
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset:
   - Place the DNS dataset file in the `data/` directory.
   - Preprocess the dataset to extract the "answer" field and generate the "formatted_answer" column.

2. Configure the models:
   - Adjust the model hyperparameters in the `test.py` script, if necessary.
   - Specify the paths to the dataset and trained models.

3. Train and evaluate the models:
   - Run the `TrainModels.py` script to train and evaluate the BERT classification models.
   - The script will output the performance metrics and save the trained models in the `models/` directory.
   - You can evaluate your models with Prediction_and_Evaluation.py

4. Perform inference:
   - Use the trained models to classify new DNS requests by providing the appropriate input data and loading the corresponding model weights.

## Results

The performance of the model was evaluated using standard classification metrics, including accuracy, precision, recall, and F1-score. The evaluation results are available in the project "classification report output.txt".

## Contributors

- David Hodefi (@hodefiDavid)
- Gal Braymok (@CodesParadox)
- Eden Dahary (@edendahary)
- Netanel Indik (@netanelin)

## Thanks

Special thanks to Pr. Amit Dvir and Dr. Enidjar Or Chaim for their valuable input and assistance throughout the project. 
We thanks Akamai Technologies, for trusting us with their valuable data, and let us explore and study the world of DNS, we appreciate it. 
 
We would also like to thank the open-source community for their continuous support.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt the project according to your needs. Contributions are also welcome.

## Contact

For any questions or inquiries, please contact:

- Gal Braymok: galbraimok@hotmail.com
- David Hodefi: davidhodefi0@gmail.com
- Eden Dahary: edendahary1@gmail.com
- Netanel Indik: netanel117@gmail.com


---

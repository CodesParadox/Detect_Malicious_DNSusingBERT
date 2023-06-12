import Load_and_Save
import Features_extraction
import TrainModels
import Prediction_and_Evaluation
from sklearn.model_selection import train_test_split
import Prediction_and_Evaluation
from sklearn.model_selection import train_test_split
import Load_and_Save
import Prediction_and_Evaluation
from sklearn.model_selection import train_test_split

# if you want to get the info in txt file change print_to_file to True
print_to_file = True
# load new dataframe (the moduls was not trained on it)
df2 = Load_and_Save.load_df("saved_data_frame/df2_include_added_features.pkl")
df2.head()


# ######################################## New classification BERT MODOL ########################################################
# making the formmated answer column - that we use for the classificatin model 
df2 = Features_extraction.get_filter_out_calculate(df2)
# get the formatted answer as a list - this list will go to bert clasification model 
the_formatted_answer_as_list = df2['formatted_ans'].tolist()
# this list is used to evaluate the prediction
true_labels = df2['is_malicious'].tolist()
# NOTE: the model was too big to upload to git, you can train the model yourself, by using the function classification_model_bert in /TrainModels.py 
evaluate_BERT_classification_model("the model" , the_formatted_answer_as_list, true_labels, print_to_file=False):


# ########################################old ANN DL MODOL ########################################################
# # load new dataframe for tha ann (the moduls was not trained on it)
# new_df2 = Load_and_Save.load_df("saved_data_frame/new_df2_bert_embading_as_columns.pkl")
# new_df2.head()

# X_train, X_test, y_train, y_test = train_test_split(new_df2, df2['is_malicious'], test_size=0.2, random_state=42)

# Prediction_and_Evaluation.ann_prediction_and_evaluation("model/ann_model_bert_embading.pkl", X_test, y_test,
#                                                         print_to_file)

# ###########################################old ML MODOL ###########################################################

# # euclidean_distance
# X_train, X_test, y_train, y_test = train_test_split(df2[["euclidean_distance_emmbading"]],
#                                                     df2['is_malicious'].values.ravel(), test_size=0.4,
#                                                     random_state=42)
# Prediction_and_Evaluation.predict_and_evaluation("model/model_random_forest_euclidean_bert_distance.sav", X_test,
#                                                  y_test, print_to_file)

# # manhattan_distance
# X_train, X_test, y_train, y_test = train_test_split(df2[["manhattan_distance"]], df2['is_malicious'], test_size=0.4,
#                                                     random_state=42)
# Prediction_and_Evaluation.predict_and_evaluation("model/model_random_forest_manhattan_bert_distance_n.sav", X_test,
#                                                  y_test, print_to_file)

# import libraries
import data_loading as dl
import data_preprocessing as dp
import model_evaluation as me

# load dataset
train_x, train_y, test_x, test_y = dl.load_dataset()
# prepare pixel data
train_x, test_x = dp.prep_pixels(train_x, test_x)
# evaluate model
scores, histories = me.evaluate_model(train_x, train_y)
# learning curves
me.summarize_diagnostics(histories)
# summarize estimated performance
me.summarize_performance(scores)


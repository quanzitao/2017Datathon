import tempfile
import urllib
import pandas as pd
import tensorflow as tf


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
       	values=df[k].astype(str).values,
		# values=df[k].values,
       	dense_shape=[df[k].size, 1])
 	    					for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

def build_estimator():
	gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
	                                                     keys=["1", "0"])
	p_AddToCart = tf.contrib.layers.sparse_column_with_keys(column_name="p_AddToCart",
	                                                     keys=["1", "0"])
	osType = tf.contrib.layers.sparse_column_with_hash_bucket(
	      "osType", hash_bucket_size=1000)
	isExclusiveMember = tf.contrib.layers.sparse_column_with_keys(column_name="isExclusiveMember",
	                                                     keys=["1", "0"])
	loggedIn = tf.contrib.layers.sparse_column_with_keys(column_name="loggedIn",
	                                                     keys=["1", "0"])
	p_MapInteraction = tf.contrib.layers.sparse_column_with_keys(column_name="p_MapInteraction",
	                                                     keys=["1", "0"])
	# Continuous base columns.
	p_sessionActivity = tf.contrib.layers.real_valued_column("p_sessionActivity")
	p_sessionDuration = tf.contrib.layers.real_valued_column("p_sessionDuration")
	p_pageViews = tf.contrib.layers.real_valued_column("p_pageViews")
	daysToCheckin = tf.contrib.layers.real_valued_column("daysToCheckin")
	daysFromPreviousVisit = tf.contrib.layers.real_valued_column("daysFromPreviousVisit")
	p_TotalPrice = tf.contrib.layers.real_valued_column("p_TotalPrice")

	model_dir = tempfile.mkdtemp()
	m = tf.contrib.learn.LinearClassifier(feature_columns=[gender,p_AddToCart,osType,
		isExclusiveMember,loggedIn,p_MapInteraction,p_sessionActivity,p_sessionDuration,
		p_pageViews,daysToCheckin,daysFromPreviousVisit,p_TotalPrice],
		model_dir=model_dir)
	
	return m


# Process input files
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("clean_training_set.csv", train_file.name)
urllib.urlretrieve("clean_test_set.csv", test_file.name)

COLUMNS = ["gender","p_sessionActivity","p_AddToCart",
			"p_sessionDuration","p_pageViews","daysToCheckin","osType",
			"daysFromPreviousVisit","p_TotalPrice","isExclusiveMember","loggedIn",
			"p_MapInteraction","BookingPurchase"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

# Create label for BookingPurchase
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["BookingPurchase"].apply(lambda x: x>0)).astype(int)
df_test[LABEL_COLUMN] = (df_test["BookingPurchase"].apply(lambda x: x>0)).astype(int)

CATEGORICAL_COLUMNS = ["gender","p_AddToCart","osType","isExclusiveMember","loggedIn",
						"p_MapInteraction"]
CONTINUOUS_COLUMNS = ["p_sessionActivity","p_sessionDuration","p_pageViews","daysToCheckin",
						"daysFromPreviousVisit","p_TotalPrice"]

# Build estimator
m = build_estimator()

# Train and evaluate
m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])

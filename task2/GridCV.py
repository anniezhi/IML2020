label='LABEL_BaseExcess'
print('-----------------------------------------------------')
print(label)
train_features = pd.read_csv('train_features_r.csv')
train_labels = pd.read_csv('train_labels.csv')

#PreProcessing

X = get_X(train_features, features)
X = zero_nan(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = get_y(train_labels, label)

#GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression(random_state = 42, class_weight='balanced')
parameters = {'max_iter':[10,100,1000]}
clf = GridSearchCV(model, parameters, scoring=('roc_auc'))
clf.fit(X_train, y_train)
pprint.pprint(clf.cv_results_)
model=clf.best_estimator_


#Fit Best Model
model.fit(X_train, y_train)
y_pred = model.decision_function(X_test); y_pred = sigmoid(y_pred)
test_result = roc_auc_score(y_test, y_pred)
print('Test Result')
print(test_result)
print('-----------------------------------------------------------')
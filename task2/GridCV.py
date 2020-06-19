model_dic = {}
param_dic = {}
labels = ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos',
          'LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2',
          'LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis']

for label in labels:
    model_dic[label] = LogisticRegression(random_state = 42, class_weight='balanced')
    param_dic[label] = {'max_iter':[10,100,1000]}

for label in labels:

    print('-----------------------------------------------------')
    print(label)
    train_features = pd.read_csv('train_features_r.csv')
    train_labels = pd.read_csv('train_labels.csv')

    X = get_X(train_features, features)
    X = zero_nan(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = get_y(train_labels, label)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = model_dic[label]
    parameters = param_dic[label]
    clf = GridSearchCV(model, parameters, scoring=('roc_auc'))
    clf.fit(X_train, y_train)
    pprint.pprint(clf.cv_results_)
    model=clf.best_estimator_
    model_dic[label] = model

    model.fit(X_train, y_train)
    y_pred = model.decision_function(X_test); y_pred = sigmoid(y_pred)
    test_result = roc_auc_score(y_test, y_pred)
    print('Test Result')
    print(test_result)
    print('-----------------------------------------------------------')

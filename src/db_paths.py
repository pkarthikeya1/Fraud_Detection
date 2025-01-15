DataBasePath: str = r"C:\Users\karthikeya\Desktop\Database_Capstones.db"
DataBaseName: str = "Database_Capstones"
SQL_QUERY: str = "SELECT * FROM Fraud_detection"


best_params = {'n_estimators': 20, 'min_samples_split': 60, 'min_samples_leaf': 100, 'max_features': 'sqrt', 'max_depth': 4, 'ccp_alpha': 0.0001, 'class_weight':'balanced'}
import numpy as np

from sklearn.linear_model import LogisticRegression

hours = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 8, 4.75, 5, 5.5]).reshape(-1, 1)

approved = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])


lr = LogisticRegression()

lr.fit(hours, approved)

new_hours = np.array([1, 5.22, 4, 3.4, 6, 0]).reshape(-1, 1)

prediction = lr.predict(new_hours)


prob_predictions = lr.predict_proba(new_hours)


np.set_printoptions(3)
print('Prediction data:')
print('New Hours:           {}'.format(new_hours.reshape(1,-1)))
print('Approved or not:     {}'.format(prediction))
print('Probability:         {}'.format(prob_predictions[:,1]))

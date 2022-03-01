from flask import Flask, render_template, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, FloatField, RadioField, SubmitField
from wtforms.validators import InputRequired
import joblib
import pandas as pd

application = Flask(__name__, template_folder='templates')
application.config['SECRET_KEY']='IDS594'

class ApplicantInfo(FlaskForm):
    universities = pd.read_csv('universities.csv', dtype='str')
    universities = [tuple(uni) for uni in universities.to_numpy()]
    university = SelectField(u'Select University: ', choices=universities)
    gre_score = IntegerField('Enter your GRE Score: ',validators=[InputRequired()])
    eng_test = RadioField('Choose English Test', choices=[('1','TOEFL'),('2','IELTS')])
    eng_test_score = FloatField('Enter your English Test Score: ',validators=[InputRequired()])
    undergrad_score = FloatField('Enter your Undergrad Score (out of 10): ',validators=[InputRequired()])
    work_ex = IntegerField('Enter your work experience (in months): ',validators=[InputRequired()])
    submit = SubmitField('Submit')

@application.route('/',methods=['GET','POST'])
def applicant_info():
    form = ApplicantInfo()
    if form.validate_on_submit():
        session['university_rank'] = form.university.data
        session['university_name'] = dict(form.university.choices).get(form.university.data)
        session['gre_score'] = form.gre_score.data
        session['eng_test'] = form.eng_test.data
        session['eng_test_score'] = form.eng_test_score.data
        session['undergrad_score'] = form.undergrad_score.data
        session['work_ex'] = form.work_ex.data

        return redirect(url_for("predict"))
    return render_template('index.html', form=form)

@application.route('/predict')
def predict():
    app_info = [[session['university_rank'], session['gre_score'], session['eng_test'], session['eng_test_score'], session['undergrad_score'], session['work_ex']]]
    app_info = pd.DataFrame(app_info, columns=['university_rank', 'gre', 'eng_test', 'eng_test_score', 'undergrad_score', 'work_ex'])
    pred = get_prediction(app_info)
    recos = get_recommendations(app_info)
    return render_template('predict.html', pred=pred, recos=recos)

def get_prediction(app_info):
    classifier = joblib.load('classifier.pkl')
    prediction = classifier.predict(app_info)
    if prediction[0]:
        return 'Admit'
    else:
        return 'Reject'

def get_recommendations(app_info):
    scalar = joblib.load('scalar.pkl')
    recommender = joblib.load('recommender.pkl')
    university_name = pd.read_csv('data.csv', usecols=['university_name'])

    distances, indices = recommender.kneighbors(scalar.transform(app_info), n_neighbors=10)
    recommendations = list(set(university_name.values[index][0] for index in indices[0]))
    
    if session['university_name'] in recommendations: 
        recommendations.remove(session['university_name'])

    if len(recommendations) >= 5:
         recommendations = recommendations[:5]

    return recommendations

if __name__ == '__main__':
    application.run(debug=True)

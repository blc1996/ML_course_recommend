from flask import Flask, render_template, request
import recommender

app = Flask(__name__)

post = [list(i) for i in recommender.run('Railroad Transportation Engrg', 'BarkanChristopherP')]
 
@app.route("/")
def index():  
    #do your things here
    return render_template('index.html')

@app.route("/", methods = ['POST'])
def test():  
    #do your things here
    course = request.form['course']
    name = request.form['prof']

    test = [list(i) for i in recommender.recommendations(course, name)]
 
    return render_template('pass.html', posts = test)
 

if __name__ == "__main__":
    app.run()


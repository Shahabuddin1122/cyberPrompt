from flask import Flask, render_template, request, jsonify, session

from machine_learning.final_response import get_final_response

app = Flask(__name__)

summary = 'no'
app.secret_key = 'your_secret_key_here'


@app.route('/')
def get_initial_page():
    return render_template('base.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    active_query = data.get('query')
    previous_query = session.get('previous_query', "")

    try:
        related, modified_query, predicted_initial_label, formatted_response, link = get_final_response(active_query,
                                                                                                        previous_query,
                                                                                                        summary)
        if related == 'yes':
            session['previous_query'] = modified_query
        else:
            session['previous_query'] = ""

        print(f"Predicted Initial Label: {predicted_initial_label}")
        print(f"Formatted Response: {formatted_response}")
        print(f"Links and Summaries: {link}")
        response = {
            'label': predicted_initial_label,
            'response': formatted_response,
            'link': link
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'label': str(e)})


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


if __name__ == '__main__':
    app.run()

from flask import Blueprint, render_template, request, jsonify, flash

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    data = request.form
    return render_template('login.html')

@auth.route('/logout', methods=['GET', 'POST'])
def logout():
    return jsonify({"message": "Logged out successfully"}), 200

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = request.form.get('user')
        password_rep = request.form.get('password_rep')
        if password != password_rep:
            return jsonify({"error": "Passwords do not match"}), 400
    return render_template('sign_up.html')
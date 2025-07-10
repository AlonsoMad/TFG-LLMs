from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, current_user, login_required

from backend.models import User
from backend import db
import re

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash("Login successful", "success")
            login_user(user, remember=True)
            return redirect(url_for('views.home'))
        else:
            flash("Login failed. Check your email and password.", "danger")
    return render_template('login.html', user=current_user)

@auth.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash("You have been logged out", "success")
    return redirect(url_for('auth.login'))

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = request.form.get('user')
        password_rep = request.form.get('password_rep')

        if len(email) < 4:
            flash("Email must be greater than 3 characters", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters", "danger")
        elif password != password_rep:
            flash("Passwords do not match", "danger")
        elif not re.search(r'[!@#$%^&*(),.?":{}|<>_+=\-\[\]\\;\'\/~`]', password):
            flash("Password must contain at least one special character", "danger")
        else:
            new_user = User(email=email, password=generate_password_hash(password, method='pbkdf2:sha256'), user=user)
            try:
                db.session.add(new_user)
                db.session.commit()
                login_user(user, remember=True)
            except Exception as e:
                db.session.rollback()
                flash(f"An error occurred while creating the account: {str(e)}", "danger")
            flash("Account created successfully", "success")
            return redirect(url_for('views.home'))

    return render_template('sign_up.html', user=current_user)
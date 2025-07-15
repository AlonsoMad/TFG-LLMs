from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, current_user, login_required

from models import User
from __init__ import db
import re

auth = Blueprint('auth', __name__)


def validate_password(password, password_rep):
    """Validate account strength."""
    veredict = (True, "Valid account.")

    if len(password) < 8:
        flash("Password must be at least 8 characters", "danger")
        veredict = (False, "Invalid password length.")
    elif password != password_rep:
        flash("Passwords do not match", "danger")
        veredict = (False, "Passwords do not match.")
    elif not re.search(r'[!@#$%^&*(),.?":{}|<>_+=\-\[\]\\;\'\/~`]', password):
        flash("Password must contain at least one special character", "danger")
        veredict = (False, "Password must contain a special character.")
    return veredict

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash("Login successful", "success")
            login_user(user, remember=True)
            print(f"User {user.email} logged in successfully")
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
        user = request.form.get('username')
        password_rep = request.form.get('password_rep')
        
        if validate_password(password, password_rep)[0] is True:
            new_user = User(email=email, password=generate_password_hash(password, method='pbkdf2:sha256'), user=user)
            try:
                db.session.add(new_user)
                db.session.commit()
                # Query the user back to confirm insertion
                confirmed_user = User.query.filter_by(email=email).first()
                if confirmed_user:
                    print(f"User {confirmed_user.email} successfully added to DB with ID {confirmed_user.id}")
                else:
                    print("User insertion failed: user not found after commit")

                login_user(new_user, remember=True)
                flash("Account created successfully", "success")
            except Exception as e:
                db.session.rollback()
                flash(f"An error occurred while creating the account: {str(e)}", "danger")
            return redirect(url_for('views.home'))

    return render_template('sign_up.html', user=current_user)
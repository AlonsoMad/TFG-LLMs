from flask import Blueprint, render_template, request,flash, jsonify
from flask_login import current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash

from models import User
from auth import validate_password
from __init__ import db

views = Blueprint('views', __name__)

@views.route('/')
# @login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/about')
def about():
    return render_template("about.html", user=current_user)

@views.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':

        new_email = request.form.get('email')
        new_username = request.form.get('username')
        new_password = request.form.get('password')
        new_password_rep = request.form.get('password_rep')
        
        update = False

        if new_email and new_email != current_user.email:
            current_user.email = new_email
            update = True
        if new_username and new_username != current_user.user:
            current_user.user = new_username
            update = True
        if new_password and new_password_rep and new_password == new_password_rep:
            valid , message = validate_password(new_password, new_password_rep)
            print(f"Password validation result: {valid}, message: {message}")
            if valid:
                current_user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                update = True
            else:
                flash(message, "danger")
                return render_template("profile.html", user=current_user)
            
        if update:
            db.session.commit()
            flash("Profile updated successfully!", "success")
        else:
            flash("No changes made to the profile.", "info")
    return render_template("profile.html", user=current_user)
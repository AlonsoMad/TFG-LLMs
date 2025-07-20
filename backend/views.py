from flask import Blueprint, render_template, request,flash, jsonify
from flask_login import current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash

from models import User
import dotenv
import os
import pandas as pd
import numpy as np
import requests
from tools.tools import load_datasets
from auth import validate_password
from __init__ import db

views = Blueprint('views', __name__)
dotenv.load_dotenv()

@views.route('/')
# @login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/about_us')
@login_required
def about_us():
    return render_template("about_us.html", user=current_user)


@views.route('/datasets')
@login_required
def datasets():
    dataset_path = os.getenv("DATASET_PATH", "/Data/3_joined_data")
    dataset_list = []
    if not os.path.exists(dataset_path):
        flash(f"Dataset path {dataset_path} does not exist.", "danger")
        return render_template("datasets.html", user=current_user, datasets=[])
    else:
        # try:
        dataset_list, datasets_name, shapes = load_datasets(dataset_path)
        if dataset_list:
            flash(f"Datasets loaded successfully!", "success")
        else:
            flash(f"No datasets found in {dataset_path}.", "warning")
        return render_template("datasets.html", user=current_user, datasets=dataset_list, names=datasets_name, shape=shapes)




@views.route('/dataset_selection', methods=['POST'])
@login_required
def dataset_selection():
    data = request.get_json()
    dataset = data.get('dataset')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"

    if not dataset:
        flash('No dataset provided', 'danger')
        return jsonify({'error': 'No dataset provided'}), 400

    print("Received dataset:", dataset)
    response = requests.post(f'{mind_api_url}/initialize', json={'dataset': dataset})
    
    return jsonify({'message': 'Dataset received', 'dataset': dataset})

@views.route('/topic_selection', methods=['GET', 'POST'])
@login_required
def topic_selection():
    data = request.get_json()
    topic_id = data.get('topic_id')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"

    if not topic_id:
        flash('No topic ID provided', 'danger')
        return jsonify({'error': 'No topic ID provided'}), 400

    print("Received topic ID:", topic_id)
    response = requests.get(f'{mind_api_url}/topic_documents', params={'topic_id': topic_id})
    
    return jsonify(response.json())

@views.route('/mode_selection', methods=['GET', 'POST'])
@login_required
def mode_selection():
    data = request.get_json()
    mode = data.get('instruction')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"
    print("Received mode:", mode)

    if not mode:
        flash('No mode provided', 'danger')
        return jsonify({'error': 'No mode provided'}), 400
    
    status_resp = requests.get(f"{mind_api_url}/status")
    status_data = status_resp.json()
    status = status_data.get("state", "unknown")
    
    if status in ["initialized", 'topic_exploration']:
        print(f"Current MIND status: {status}")
        if mode == "Explore topics":
            response = requests.get(f'{mind_api_url}/explore')
            return jsonify(response.json())
        
        elif mode == "Analyze contradictions":
            response = requests.post(f'{mind_api_url}/explore')
            return jsonify(response.json())
        
        else:
            flash('Invalid mode selected', 'danger')
            return jsonify({'error': 'Invalid mode selected'}), 400
    else:
        flash(f"Select a dataset before exploring!", "warning")
        return jsonify({'error': f'MIND is not initialized. Current status: {status}'})

@views.route('/detection', methods=['GET', 'POST'])
@login_required
def detection():
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"
    dataset_path = os.getenv("DATASET_PATH", "/Data/3_joined_data")
    status = "idle"
    mind_info = {}

    try:
        data = request.get_json()
        dataset = data.get('dataset', None) if data else None
        print(data)
    except Exception as e:
        dataset = None
    try:
        # Check current MIND status
        status_resp = requests.get(f"{mind_api_url}/status")
        status_data = status_resp.json()
        status = status_data.get("state", "unknown")
        print(status)

        if status in ["idle", "failed", "completed"]:
            # init_resp = requests.post(f"{mind_api_url}/initialize")
            # flash(init_resp.json().get("message"), "info")
            ds_tuple = load_datasets(dataset_path)
            if ds_tuple[0]:
                flash("Datasets loaded successfully!", "success")
                flash("MIND is idle, please wait...", "warning")

            else:
                flash("No datasets found or error loading datasets.", "warning")

            if dataset:
                response = requests.post(f'{mind_api_url}/initialize', json={'dataset': dataset})

        elif status == "initializing":

            ds_tuple = ( [], [], [])
            flash("MIND is initializing, please wait...", "warning")
        elif status == "initialized":
            ds_tuple = load_datasets(dataset_path)
            flash("MIND already initialized.", "success")
        elif status == "topic_exploration":
            ds_tuple = ( [], [], [])
            try:
                explore_resp = requests.get(f'{mind_api_url}/explore')
                if explore_resp.status_code == 200:
                    mind_info = explore_resp.json().get("topic_information", {})
                else:
                    flash("Failed to explore topics.", "warning")
            except Exception as e:
                flash(f"Error contacting MIND: {str(e)}", "danger")
        elif status == "running":

            flash("MIND is currently processing.", "info")
        elif status == "completed":

            flash("MIND processing completed.", "success")
        else:
            flash("Unknown MIND state.", "danger")

    except requests.RequestException as e:
        flash(f"Error connecting to MIND: {e}", "danger")

    return render_template("detection.html", user=current_user, status=status, ds_tuple=ds_tuple, mind_info=mind_info)


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
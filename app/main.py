from flask import send_from_directory, render_template, url_for, request, redirect, session, jsonify, make_response
from app import webapp


import requests


@webapp.route('/')
def main():
    return render_template("pages/main.html")

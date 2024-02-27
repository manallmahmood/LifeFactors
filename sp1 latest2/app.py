from flask import Flask

# Create a new Flask app
app = Flask(__name__)

# Import route definitions from diabetesapp.py
from appdiabetes import *

# Import route definitions from appheart.py
from appHeart import *

# Import route definitions from appheartblood.py
from appheartblood import *

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)

from main import app

if __name__ == '__main__':
    print("Starting Used Bike Price Predictor...")
    print("Visit http://127.0.0.1:5000 to access the application")
    app.run(debug=True, host='127.0.0.1', port=5000)
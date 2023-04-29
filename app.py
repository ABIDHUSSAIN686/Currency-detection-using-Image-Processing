from flask import *
import othermodule
app=Flask(__name__)



@app.route("/")
def upload():
    return render_template("file_upload.html")


@app.route("/", methods=["POST"])
def compare():
    file1 = request.files['image1']
    filename1 = file1.filename
    data1 = file1.read()  # read the image data from the file
    size1 = len(file1.read())
    # Finding the Front-Page Keypoints
    result = othermodule.mainfunction(data1)
    return jsonify({
        "filename1": filename1,
        "size1": len(data1),
        "result":result
         # include the result from the other function in the response
    })

if __name__ == "__main__":
    app.run(debug=True)

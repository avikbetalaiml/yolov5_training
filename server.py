
from flask import Flask, request
from yolo5 import yolov5_training


app = Flask(__name__)


@app.route('/compare_train', methods = ['POST','GET'])

def compare_train():
	values = {}
	epochs = request.args.get('epochs')
	frame_work = request.args.get('frame_work')
	algo = request.args.get('algo')
	project_id = request.args.get('project_id')
	#if (epochs, frame_work, algo, project_id) not in None:
	if algo =='yolov5' and frame_work == 'pytorch':
		yolov5obj = yolov5_training(project_id=project_id, epochs=epochs, frame_work=frame_work, algo=algo)
		get_train = yolov5obj.mlflow_init()
		values = {
			"epochs": epochs,
			"frame_work":frame_work,
			"algo":algo,
			"project_id": project_id
		}
	return values

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(port = 5002)

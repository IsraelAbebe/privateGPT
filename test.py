from gradio_client import Client

client = Client("http://127.0.0.1:8001/")
result = client.predict(
		["test.jpg"],	# List[filepath]  in 'Upload File(s)' Uploadbutton component
		api_name="/_upload_file"
)
print(result)
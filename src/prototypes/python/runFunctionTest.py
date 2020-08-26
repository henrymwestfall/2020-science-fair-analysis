def deco(func):
	print("Decorations!")
	func()

@deco # runs on definition
def run():
	print("Hello world!")


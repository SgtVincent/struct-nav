python3 -m black aslam
python3 -m isort aslam
python3 -m pylint --ignored-modules=cv2,open3d aslam
python3 -m pydocstyle --convention=google aslam
# python3 -m mypy --install-types --strict aslam

python3 -m black habitat
python3 -m isort habitat
python3 -m pylint --ignored-modules=cv2,open3d habitat/scripts
python3 -m pydocstyle --convention=google habitat
# python3 -m mypy --install-types --strict habitat

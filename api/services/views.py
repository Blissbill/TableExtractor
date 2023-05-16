from flask import Blueprint, jsonify

blueprint = Blueprint('services', __name__)


@blueprint.route('/health', methods=['GET'])
def health():
    return jsonify("OK")
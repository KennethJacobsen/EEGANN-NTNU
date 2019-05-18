"""
Emotiv cortex communications

Created 25. February 2019

@author: KSVJ
"""

# EmotivCortex related imports
import asyncio
import websockets as websockets
import json as json
import ssl
import string
import secrets


def init():
    global _authToken, _ssl_context, _url, _websocket
    # Emotiv constants
    _authToken = ''
    _ssl_context = ssl._create_unverified_context()
    _url = 'wss://emotivcortex.com:54321'
    # Connect to websocket
    _websocket = asyncio.get_event_loop().run_until_complete(websockets.connect(_url, ssl=_ssl_context))

# --------------------------------------------DEFAULT USER--------------------------------------------
# _clientID = ''
# _clientSecret= ''
# _uName = ''
# _pWord = ''


# Update default data
def updateInfo(clientID, clientSecret, uName, pWord):
    global _clientID, _clientSecret, _uName, _pWord
    _clientID, _clientSecret, _uName, _pWord = clientID, clientSecret, uName, pWord

# ---------------------------------------------Help methods------------------------------------------------------------
# Check if response returned an error, and return error msg
def _checkError(d):
    response = ''
    boolean = False
    for i in range(0, len(d)):
        key = 'error'
        if key in d:
            response = d[key]
            boolean = True
    return boolean, response

# Checks for key in dict of kwargs
def _checkKwords(key, data):
    for i in range(0, len(data)):
        if key in data:
            return True
        else:
            return False

# Send data to websocket
async def _sendData(data):
    global _websocket
    await _websocket.send(data)

# Receive data from websocket
async def _recvData():
    global _websocket
    return json.loads(await _websocket.recv())

# Websocket communication
def _socketCommunication(data):
    asyncio.get_event_loop().run_until_complete(_sendData(data))
    return asyncio.get_event_loop().run_until_complete(_recvData())

# return errormsg or result
def _getReturn(error, msg, response, auth=False):
    global _authToken
    if error:
        return msg
    elif _checkKwords('result', response):
        if auth:
            _authToken = response['result']['_auth']
            return response['result']['_auth']
        else:
            return response['result']
    else:
        return response

# Generate random ID
def _generateID(size=3, chars=string.ascii_letters+string.digits):
    return ''.join(secrets.choice(chars) for _ in range(size))

# Check if ID matches
def _checkID(sendDict, response):
    if sendDict['id'] == response['id']:
        return True
    else:
        return False

# Create dict
def _createDict(method, params=None):
    return json.dumps({'jsonrpc': '2.0', 'method': method, 'params': params, 'id': _generateID()})


# Handles data, creating json, then checking response, id and error.
def _dataHandling(sendDict, auth=False):
    response = _socketCommunication(sendDict)
    if _checkID(json.loads(sendDict), response):
        error, msg = _checkError(response)
        return _getReturn(error, msg, response, auth)
    else:
        return 'ID does not match.'

# ----------------------------------------------Cortex methods---------------------------------------------------------
# Login to specified user
def login(user, password, ID, secret):
    method = 'login'
    params = {'username': user, 'password': password, 'client_id': ID, 'client_secret': secret}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Login to default user
def defaultLogin():
    method = 'login'
    params = {'username': _uName, 'password': _pWord, 'client_id': _clientID, 'client_secret': _clientSecret}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Get username of user logged in
# @return: list of users logged in
def getUserLogin():
    method = 'getUserLogin'
    sendDict = _createDict(method)
    return _dataHandling(sendDict)

# Logout of desired user
def logout(user):
    method = 'logout'
    params = {'username': user}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Logout of default user
def defaultLogout():
    method = 'logout'
    params = {'username': _uName}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Authenticate a specified user or as anonymous.
# If not anonymous: 'ID' = client_id and 'secret' = client_secret
def authorize(anonymous=False, **kwargs):
    method = 'authorize'
    params = {}
    if anonymous is False:
        params['client_id'] = kwargs['ID']
        params['client_secret'] = kwargs['secret']
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict, auth=True)

# Authenticate default user
def defaultAuthorize():
    method = 'authorize'
    params = {'client_id': _clientID, 'client_secret': _clientSecret, 'debit': 1000}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict, auth=True)

# Generate new token
def generateNewToken():
    method = 'generateNewToken'
    params = {'token': _authToken}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict, auth=True)

# Accept the license. End user has to accept license to use cortex.
def acceptLicense():
    method = 'acceptLicense'
    params = {'_auth': _authToken}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict, auth=True)

# Query which headsets are connected
def queryHeadsets():
    method = 'queryHeadsets'
    sendDict = _createDict(method)
    return _dataHandling(sendDict)

# Create a new session. Sessions are used to manage live or pre-recorded data from the headset.
# Can be created using only status, using status and headset id, and status, headset id and sensors.
# @return: dictionary with info
def createSession(status, **kwargs):
    method = 'createSession'
    params = {'_auth': _authToken, 'status': status}
    if _checkKwords('headset', kwargs):
        params['headset'] = kwargs['headset']
    if _checkKwords('sensors', kwargs):
        params['sensors'] = kwargs['sensors']
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Query all sessions. @return list of dictionaries, where dictionary at index 0 = session 1.
def querySessions():
    method = 'querySessions'
    params = {'_auth': _authToken}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Update an existing session. Sessions are used to manage live or pre-recorded data from the headset.
# Can be created using only status, or using status and any or all of the following, headset, sensors
# and session id.
# @return: dictionary with info
def updateSession(status, **kwargs):
    method = 'updateSession'
    params = {'_auth': _authToken}
    if _checkKwords('session', kwargs):
        params['session'] = kwargs['session']
    params['status'] = status
    if _checkKwords('sensors', kwargs):
        params['sensors'] = kwargs['sensors']
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Subscribe to a stream of data from a headset.
# @input: streams = array of streams to subscribe to, session = session id to subscribe to
# if session is not set cortex will subscribe to it's first session.
# @return: list of dictionaries
def subscribe(streams, session=''):
    method = 'subscribe'
    params = {'_auth': _authToken, 'streams': streams, 'session': session}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Unsubscribe to (a) stream(s) of data from a headset
# @input: streams = array of stream(s) to unsubscribe to, session = session id to unsubscribe to
# if session is not set cortex will unsubscribe to it's first session
# @return: list of dictionaries with key: "message"
def unsubscribe(streams, session=''):
    method = 'unsubscribe'
    params = {'_auth': _authToken, 'streams': streams, 'session': session}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Create and save mapping channel for Flex headset
# @input: status = string-operation, can be "get", "create", "read", update" or "delete", if not set left empty
# @input: uuid = string-uuid of config. Set this value when read/update/delete config, if not set left empty
# @input: name = string-name of config. Set this value when create/update config, if not set left empty
# @input: mapping = json object, mapping of value of config. Set when create/update config,
# if not set left empty
# @return: string with message
def configMapping(status='', uuid='', name='', mapping=None):
    method = 'configMapping'
    params = {'_auth': _authToken, 'status': status, 'uuid': uuid, 'name': name, 'mapping': mapping}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Get info on the current license.
def getLicenseInfo():
    method = 'getLicenseInfo'
    params = {'_auth': _authToken}
    sendDict = _createDict(method, params)
    return _dataHandling(sendDict)

# Unsubscribe without return.
# @args: tries = number of attempts to unsubscribe before giving up. 20 by default.
def unsubscribeNoRet(streams, tries=20, session=''):
    method = 'unsubscribe'
    params = {'_auth': _authToken, 'streams': streams, 'session': session}
    sendDict = _createDict(method, params)
    response = str(_socketCommunication(sendDict))
    x = 0
    while 'success' not in response:
        response = _socketCommunication(sendDict)
        x += 1
        if x > tries:
            return
    return response

# Close a session, no return.
def closeSession(**kwargs):
    method = 'updateSession'
    params = {'_auth': _authToken, 'status': 'close'}
    if _checkKwords('session', kwargs):
        params['session'] = kwargs['session']
    sendDict = _createDict(method, params)
    response = str(_socketCommunication(sendDict))
    while not _checkKwords('error', response) and not _checkKwords('closed', response):
        response = str(_socketCommunication(sendDict))
    return response

# Receive data from cortex
# @return: data from cortex, depending on what stream is subscribed to
def receiveData():
    response = asyncio.get_event_loop().run_until_complete(_recvData())
    error, msg = _checkError(response)
    return _getReturn(error, msg, response)

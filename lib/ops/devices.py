import tensorflow as tf


def list_devices(session=None):
    if session is None:
        session = session or tf.get_default_session()
    return session.list_devices()


def list_gpu_devices(session=None):
    return [x for x in list_devices(session) if x.device_type == 'GPU']


def have_gpu():
    return len(list_gpu_devices()) != 0

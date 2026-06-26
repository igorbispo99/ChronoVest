from chronovest.data._yf_session import make_impersonated_session


def test_session_helper_never_raises():
    # returns a session if curl_cffi is installed, else None; must not raise
    sess = make_impersonated_session()
    assert sess is None or hasattr(sess, "get")





def sanitize_string(string):
    """
    Sanitize a string by removing all non-alphanumeric characters and converting it to lowercase. Also substitute spaces with _
    """
    return "".join([c if c.isalnum() else "_" for c in string.lower()]).strip()

def unsanitize_string(string):
    """
    Unsanitize a string by replacing _ with spaces
    """
    return string.replace("_", " ").strip()
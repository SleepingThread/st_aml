

class MinErrorSelect(object):
    """
    Note
    ----

    """
    def __init__(self,max_buf_size):
        self.max_buf_size = max_buf_size
        return
    
    def select(self,buf):
        buf.sort(cmp=lambda x,y: int(x["error"]>y["error"])-int(x["error"]<y["error"]))
        return buf[:self.max_buf_size]

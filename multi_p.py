import queue
import threading
import time


class Monitor:
    def __init__(self):
        self.var1 = 0
        self.var2 = 0
        # other internal variables

    def A(self):
        self.var1 += 2
        self.var2 -= 3
        time.sleep(0.1)
        print("A", self.var1, self.var2)
        pass

    def B(self):
        # 访问和修改内部变量...
        result = self.var1 + self.var2
        print("B", result)
        return result


class Manager:
    def __init__(self, monitor):
        self.monitor = monitor
        self.lock = threading.Lock()
        self.m_queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def A(self):
        self.m_queue.put('A')

    def B(self):
        self.m_queue.put('B')

    def _run(self):
        while True:
            request = self.m_queue.get(block=True)
            if request == 'A':
                with self.lock:
                    self.monitor.A()
            elif request == 'B':
                with self.lock:
                    self.monitor.B()
                return


if __name__ == '__main__':
    monitor = Monitor()
    manager = Manager(monitor)
    for i in range(10):
        manager.A()
    manager.B()

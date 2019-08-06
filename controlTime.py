import time


class Timer:
    def __init__(self):
        self.max = None
        self.min = None
        self.count = 0
        self.sum = 0
        self.mean = 0
        self.time1 = 0
        self.time2 = 0

    def start(self):
        self.time1 = time.time()

    def stop(self):
        self.time2 = time.time()

        dif = self.time2 - self.time1

        if self.max is None or dif > self.max:
            self.max = dif

        if self.min is None or dif < self.min:
            self.min = dif

        self.count += 1
        self.sum += dif
        self.mean = self.sum / self.count


def testTimer():
    t = Timer()
    t.start()
    t.stop()
    print(t.mean)
    print(t.sum)
    print(t.count)
    print(t.min)
    print(t.max)

    t.start()
    time.sleep(5)
    t.stop()
    print(t.mean)
    print(t.sum)
    print(t.count)
    print(t.min)
    print(t.max)

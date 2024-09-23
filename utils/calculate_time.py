import time

def splitting_time_eachnum(times, num, key_word, change_word):
    new_times = times.copy()
    for k, v in times.items():
        if key_word in k:
            key_change = k.replace(key_word, change_word)
            new_times[key_change] = v / num
    return new_times


class TimeCalculator:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.durations = {}

    def start(self, label):
        """记录代码段开始运行的时间点"""
        self.start_times[label] = time.time()

    def init_onetime(self, label):
        self.durations[label] = 0

    def time_add(self, label, time_cost):
        try:
            self.durations[label] += time_cost
        except KeyError:
            raise ValueError(f"未找到标签 '{label}' 的开始时间")
    def end(self, label):
        """记录代码段结束运行的时间点并计算运行时间"""
        try:
            self.end_times[label] = time.time()
            duration = self.end_times[label] - self.start_times[label]
            self.durations[label] = self.durations.get(label, 0) + duration
            return duration
        except KeyError:
            raise ValueError(f"未找到标签 '{label}' 的开始时间")

    def get_duration(self, label):
        """获取指定标签的代码段运行时间"""
        if label not in self.durations:
            raise ValueError(f"未找到标签 '{label}' 的运行时间")
        return self.durations[label]

    def splitting_time_eachnum(self, num, key_word, change_word):
        new_times = self.durations.copy()
        for k, v in self.durations.items():
            if key_word in k:
                key_change = k.replace(key_word, change_word)
                new_times[key_change] = v / num
        self.durations = new_times
        return new_times

    def print_all_durations(self):
        """打印所有记录的代码段运行时间"""
        for label, duration in self.durations.items():
            print(f"{label}: {duration:.4f} 秒")

    # 使用示例：
    # timer = TimeCalculator()
    # timer.start("代码段1")
    # # 运行一些代码
    # timer.end("代码段1")
    # print(f"代码段1运行时间: {timer.get_duration('代码段1'):.4f} 秒")
    # timer.print_all_durations()

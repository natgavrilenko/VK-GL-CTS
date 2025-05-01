import argparse
import glob
import os
import re


class Analyser:
    RE_CONSTANT = re.compile(r'(%\w+) += OpConstant %\w+ (\d+)')
    RE = {
        'OpControlBarrier': re.compile(r'OpControlBarrier %\w+ %\w+ (%\w+)'),
        'OpMemoryBarrier': re.compile(r'OpMemoryBarrier %\w+ (%\w+)'),
        'OpMemoryNamedBarrier': re.compile(r'OpMemoryNamedBarrier %\w+ %\w+ (%\w+)'),
        'OpAtomicLoad': re.compile(r'%\w+ += OpAtomicLoad %\w+ %\w+ %\w+ (%\w+)'),
        'OpAtomicStore': re.compile(r'OpAtomicStore %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicExchange': re.compile(r'%\w+ += OpAtomicExchange %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicIIncrement': re.compile(r'%\w+ += OpAtomicIIncrement %\w+ %\w+ %\w+ (%\w+)'),
        'OpAtomicIDecrement': re.compile(r'%\w+ += OpAtomicIDecrement %\w+ %\w+ %\w+ (%\w+)'),
        'OpAtomicIAdd': re.compile(r'%\w+ += OpAtomicIAdd %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicISub': re.compile(r'%\w+ += OpAtomicISub %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicSMin': re.compile(r'%\w+ += OpAtomicSMin %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicUMin': re.compile(r'%\w+ += OpAtomicUMin %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicSMax': re.compile(r'%\w+ += OpAtomicSMax %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicUMax': re.compile(r'%\w+ += OpAtomicUMax %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicAnd': re.compile(r'%\w+ += OpAtomicAnd %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicOr': re.compile(r'%\w+ += OpAtomicOr %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicXor': re.compile(r'%\w+ += OpAtomicXor %\w+ %\w+ %\w+ (%\w+) %\w+'),
        'OpAtomicFlagTestAndSet': re.compile(r'%\w+ += OpAtomicFlagTestAndSet %\w+ %\w+ %\w+ (%\w+)'),
        'OpAtomicFlagClear': re.compile(r'OpAtomicFlagClear %\w+ %\w+ (%\w+)'),
        'OpAtomicCompareExchange': re.compile(r'%\w+ += OpAtomicCompareExchange %\w+ %\w+ %\w+ (%\w+) (%\w+) %\w+ %\w+'),
        'OpAtomicCompareExchangeWeak': re.compile(r'%\w+ += OpAtomicCompareExchangeWeak %\w+ %\w+ %\w+ (%\w+) (%\w+) %\w+ %\w+')
    }

    def __init__(self, logs):
        if not os.path.exists(logs) or not os.access(logs, os.R_OK):
            raise RuntimeError('Cannot open file \'%s\'', logs)
        self.logs = logs
        self.output = 'output.log'
        if os.path.exists(self.output):
            os.remove(self.output)
        self.summary = {}

    def analyse(self):
        for file in glob.iglob(os.path.join(self.logs, '**'), recursive=True):
            if os.path.isfile(file):
                with open(file, 'r', errors='replace') as f:
                    name = None
                    data = None
                    for line in f:
                        if line.startswith('#beginTestCaseResult'):
                            name = line.split(' ')[1].strip()
                        elif '<SpirVAssemblySource>' in line:
                            data = []
                        elif '</SpirVAssemblySource>' in line:
                            self.analyse_test(name, data)
                            data = None
                        elif data is not None:
                            data.append(line)
                    if data is not None:
                        print('Test suit did not terminate correctly: ' + file)
        self.print_summary()

    def print_summary(self):
        for k, v in sorted(self.summary.items()):
            print(k + ' ' + str(sorted(v)))

    def analyse_test(self, name, data):
        result = {}
        data = '\n'.join(data)
        instructions = self.collect_instructions(data)
        if instructions:
            constants = self.extract_constants(instructions)
            values = self.collect_semantics(data, constants)
            for k, v in instructions.items():
                result[k] = []
                if k == 'OpAtomicCompareExchange' or k == 'OpAtomicCompareExchangeWeak':
                    for constant in v:
                        parts = constant.split('-')
                        result[k].append(values[parts[0]] + '-' + values[parts[1]])
                else:
                    for constant in v:
                        result[k].append(values[constant])
            self.append_summary(result)
            self.write_result(name, result)

    def collect_instructions(self, data):
        result = {}
        for k, v in self.RE.items():
            for match in re.finditer(v, data):
                if k not in result:
                    result[k] = []
                if k == 'OpAtomicCompareExchange' or k == 'OpAtomicCompareExchangeWeak':
                    result[k].append(match.group(1) + '-' + match.group(2))
                else:
                    result[k].append(match.group(1))
        return result

    def extract_constants(self, instructions):
        result = set()
        for k, v in instructions.items():
            if k == 'OpAtomicCompareExchange' or k == 'OpAtomicCompareExchangeWeak':
                for value in v:
                    result.update(value.split('-'))
            else:
                result.update(v)
        return result

    def collect_semantics(self, data, constants):
        result = {}
        for match in re.finditer(self.RE_CONSTANT, data):
            if match.group(1) in constants:
                result[match.group(1)] = match.group(2)
        for constant in constants:
            if constant not in result.keys():
                raise RuntimeError('Cannot find constant ' + constant)
        return result

    def append_summary(self, result):
        for k, v in result.items():
            for value in v:
                if value not in self.summary:
                    self.summary[value] = set()
                self.summary[value].add(k)

    def write_result(self, name, result):
        if result:
            with open(self.output, "a") as f:
                for k, v in result.items():
                    for value in v:
                        f.write(name + ',' + k + ',' + value + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs')
    args = parser.parse_args()
    analyser = Analyser(args.logs)
    analyser.analyse()


if __name__ == '__main__':
    main()
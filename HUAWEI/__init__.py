import time


def use_logging(func):
  def _deco():
    start = time.time()
    func()
    dur = time.time() - start

    if dur < 60:
        print('耗时: %.6fs'%dur)
    else:
        mins = int(dur // 60)
        sec = dur % 60
        print('耗时: {}分{:.3f}秒'.format(mins,sec))

  return _deco



@use_logging
def bar():
  print('i am bar')
bar()



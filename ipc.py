import socket
from statistics import mean

PORT = 6666


def print_results(results):
    for e in range(1, len(results) + 1):
        print('Iteration {:>2}| MSE {:>6.4f} | MAE {:>6.4f}'.format(e, results[e - 1][0], results[e - 1][1]))
    final_mse = float(mean([float(r[0]) for r in results]))
    final_mae = float(mean([float(r[1]) for r in results]))
    print('Mean        | MSE {:>6.4f} | MAE {:>6.4f}'.format(final_mse, final_mae))


def resultServer(args, q=None):
    results = []
    partials = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', PORT))
        s.listen()
        while len(results) < args.exps:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = conn.recv(4 * 1024)
                res = data.decode().split(';')
                if len(res) == 2:
                    print('\033[94mReceived result:', data,'\033[0m' )
                    results.append([float(res[0]), float(res[1])])
                    print_results(results)
                else:
                    print('\033[94mReceived training result:', data, '\033[0m')
                    it, mse, mae = int(res[0]), float(res[1]), float(res[2])
                    if it > len (partials):
                        partials.append([])
                    partials[it-1].append([mse, mae])
        s.shutdown(1)
        s.close()
    for e in range(1, args.exps + 1):
        print('Iteration {:>2}| MSE {:>6.4f} | MAE {:>6.4f}'.format(e, results[e - 1][0], results[e - 1][1]))
    final_mse = float(mean([float(r[0]) for r in results]))
    final_mae = float(mean([float(r[1]) for r in results]))
    print('Mean        | MSE {:>6.4f} | MAE {:>6.4f}'.format(final_mse, final_mae))
    print(partials)
    if q:
        q.put({"mse" : final_mse, "mae": final_mae})


def sendResults(mse, mae):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
        c.connect(('localhost', PORT))
        c.send("{:.5f};{:.5f}".format(float(mse), float(mae)).encode())
        c.close()


def sendPartials(it, mse, mae):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
        c.connect(('localhost', PORT))
        c.send("{};{:.5f};{:.5f}".format(it, float(mse), float(mae)).encode())
        c.close()
from src.search import search_prompt
import argparse
import sys

def _ask_once(pergunta: str) -> int:
    try:
        resposta = search_prompt(pergunta=pergunta)
        if not resposta:
            print("Não foi possível obter uma resposta.")
            return 2
        print(resposta)
        return 0
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        return 130
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        prog="chat",
        description="CLI para realizar perguntas ao mecanismo de busca/LLM."
    )
    parser.add_argument(
        "-q", "--question",
        help="Pergunta a ser enviada. Se omitido, inicia modo interativo quando em TTY."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Força iniciar em modo interativo (REPL)."
    )

    args = parser.parse_args()

    if args.interactive or (not args.question and sys.stdin.isatty()):
        print("Modo interativo. Use Ctrl+C ou Ctrl+D para sair.")
        print("Faça sua pergunta:")
        try:
            while True:
                try:
                    pergunta = input("Pergunta> ").strip()
                except EOFError:
                    print()
                    break
                if not pergunta:
                    continue
                code = _ask_once(pergunta)
                if code not in (0, 2):
                    return code
        except KeyboardInterrupt:
            print("\nEncerrando...")
            return 130
        return 0

    if args.question:
        return _ask_once(args.question)

    if not sys.stdin.isatty():
        pergunta = sys.stdin.read().strip()
        if pergunta:
            return _ask_once(pergunta)

    print("Nenhuma pergunta fornecida. Use -q/--question, --interactive ou forneça via stdin.")
    return 64

if __name__ == "__main__":
    sys.exit(main())
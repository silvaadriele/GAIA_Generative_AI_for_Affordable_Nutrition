import json
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_para_perfil.json>")
        print("Exemplo de perfil:")
        print(json.dumps({
            "age_months": 24,
            "sex": "M",
            "weight_kg": 10.5,
            "height_cm": 82.0,
            "region": "Sul"
        }, indent=2))
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        profile = json.load(f)

    from orchestrator.gaia import GAIA
    result = GAIA().run(profile)
    paths = result.save()
    print(f"\nRelatório salvo em:\n  {paths['txt']}\n  {paths['json']}")


if __name__ == "__main__":
    main()

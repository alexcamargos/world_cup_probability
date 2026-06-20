from main import main


def test_main_prints_banner(capsys) -> None:
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello from world-cup-probability!"

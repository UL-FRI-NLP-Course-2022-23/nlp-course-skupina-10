# BART-Paraphrase Model

Based on <https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing>

```sh
ulimit -n 64000
python3 train.py
```

## Example

input:
```
Z Davidom Juričem, predsednikom sekcije za osebna vozila pri Gospodarski zbornici Slovenije in generalnim direktorjem Summit Motors, ki zastopa Ford, smo se pogovarjali o razlogih za to odločitev in njenem pomenu za Slovenijo, pa tudi o novem merjenju emisij in njegovem vplivu na davke.
```

output:
```
- Govorimo o tem in o novem merjenju emisij in njegovih vplivih na davkov Davida Juriča, predsednika sekcije za poklicna vozila pri GZS ter direktorja družbe Summit Motors, ki zastopa Ford.
- Z Davidom Juričem, predsednikom sekcije za poklicna vozila pri GZS, in generalnim direktorjem Summit Motors, ki zastopa Ford, smo se pogovarjali o razlogih za odločitev in pomenu za Slovenijo, in o novem merjenju emisij in njegovem vplivu na davki.
- O tem, zakaj jo je Ford izbral, in o tem, kakšno vlogo ta sklepstavlja za Slovenijo, ter o novem merjenju emisij in njegovem vplivu na davke smo se pogovarjali z Davidom Juričem, predsednikom sekcije za osebna vozila pri slovenskem Gospodarski zbornici in predsednikom uprave Summit Motors Group.
```

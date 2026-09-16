# Onyxia — remettre une instance en état de marche

*Giulio · v2 du 16/09/2026 · à relire au lancement de chaque service SSP Cloud*

> Les valeurs du projet sont renseignées dans `scripts/onyxia_bootstrap.sh` :
> dépôt `fogatogl/diffusion-models`, bucket MinIO `gfogato`.
> **Le bucket ne s'appelle pas `$USERNAME`** : dans le service, `$USERNAME` vaut
> `onyxia`. Les scripts qui en faisaient leur défaut visaient un bucket vide.

---

## 0. Version courte

Le formulaire Onyxia clone le dépôt tout seul (onglet Git). Il ne reste qu'une commande dans le terminal VS Code :

```bash
bash ~/work/diffusion-models/scripts/onyxia_bootstrap.sh && source ~/.bashrc
```

Si `~/work/diffusion-models` n'existe pas — volume perdu, ou onglet Git laissé vide :

```bash
cd ~/work
git clone https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/fogatogl/diffusion-models.git
bash ~/work/diffusion-models/scripts/onyxia_bootstrap.sh && source ~/.bashrc
```

Le bootstrap est **idempotent** : on peut le relancer autant de fois qu'on veut, il ne casse rien.

---

## 1. Ce qui survit et ce qui meurt

C'est toute la logique de la procédure. Un service Onyxia est un conteneur éphémère ; seul le volume persistant traverse les redémarrages.

| Chemin | Survit ? | Conséquence pratique |
|---|---|---|
| `/home/onyxia/work` | **oui**, si la persistance est cochée au lancement | venv, dépôt, artefacts lourds vivent ici |
| `~/.bashrc` | non | l'activation auto du venv est à réécrire à chaque fois |
| `~/.local` | non | le kernel Jupyter est à réenregistrer à chaque fois |
| `~/.gitconfig` | non | nom / e-mail de commit à reposer (sauf si l'onglet Git est rempli) |
| `~/.mc/config.json` | régénéré au lancement | l'alias `s3` est prêt, jeton valide **7 jours** |
| paquets `apt`, `/tmp`, tout le reste | non | ne rien y installer d'important |

**Règle unique** : rien d'important hors de `~/work`, rien de lourd hors de S3 et de Git.

---

## 2. À faire une seule fois (compte Datalab)

1. Générer un jeton GitHub sur <https://github.com/settings/tokens> — scope `repo`, expiration courte (30 j).
2. L'enregistrer dans le compte Datalab, section **Services externes**. Il devient alors `$GIT_PERSONAL_ACCESS_TOKEN` dans tous les services.
3. Vérifier que l'**e-mail du compte Datalab est celui du compte GitHub** : sinon les commits ne sont pas rattachés à ton profil GitHub — et le profil GitHub est une pièce du dossier de stage.
4. Renouveler le jeton quand il expire (note-toi la date : 30 jours, c'est court, et ça tombera en pleine S7).
5. Une fois le formulaire de service rempli correctement, **sauvegarder la configuration** (icône marque-page en haut à droite) : les lancements suivants se font en un clic.

Doc de référence : <https://docs.sspcloud.fr/content/version-control.html>

---

## 3. À chaque lancement : le formulaire

| Onglet | Réglage | Pourquoi |
|---|---|---|
| **Git** | URL complète du dépôt, ex. `https://github.com/<owner>/<repo>` | cloné automatiquement dans l'espace de travail |
| **Init** | URL publique de `onyxia_init.sh` (facultatif si tu utilises le bootstrap) | exécuté **en tant que root** juste après le démarrage |
| **Ressources** | GPU **seulement quand il en faut** | Module 0 (log-poids, resamplers, ESS) est du CPU pur : ne mobilise pas une T4 partagée pour lancer `pytest` |
| **Persistance** | activée | sinon tu réinstalles le venv à chaque session |
| **Sécurité** | changer le mot de passe du service | il s'affiche en clair sinon |

Le script d'init tourne en root, d'où le `chown -R onyxia:users` final de ton `onyxia_init.sh` — sans lui, tout ce qu'il crée t'est inaccessible.

---

## 4. Le bootstrap (`scripts/onyxia_bootstrap.sh`, versionné dans le dépôt)

Il vit **dans le dépôt**, pas sur le volume : comme l'onglet Git clone le dépôt au démarrage, le script est toujours là, même si le volume a été perdu.

**Le script fait foi** — il n'est plus recopié ici, une copie dérive toujours de l'original. Lis-le : `scripts/onyxia_bootstrap.sh`. Il fait sept choses, dans cet ordre :

1. clone le dépôt, ou `pull --ff-only` s'il est déjà là ;
2. pose l'identité git **sans écraser** celle qu'Onyxia a déjà mise (l'onglet Git la renseigne : l'écraser avec un `GIT_MAIL` en dur détacherait les commits du profil GitHub) ;
3. crée `~/work/.venvs/ddpm` en `--system-site-packages` et installe `requirements-onyxia.txt` ;
4. réenregistre le kernel Jupyter dans `~/.local` ;
5. réécrit `~/.bashrc` : `BUCKET`, `REPO`, activation du venv ;
6. rapatrie de S3 les **deux derniers checkpoints** (257 Mo) — `FULL_WEIGHTS=1` pour les 2 Go de checkpoints intermédiaires, `NO_S3=1` pour tout sauter ;
7. affiche la version de torch et le GPU vu.

Deux pièges qui y sont commentés, et qui ont coûté du temps :

- `mc ls s3/` **sans nom de bucket reste bloqué** jusqu'au timeout — la politique `stsonly` n'autorise pas `ListBuckets`. Viser toujours `s3/gfogato/...` explicitement. Un `mc` qui « ne répond pas » n'est donc pas forcément un jeton expiré.
- `particles` 0.4 déclare `numpy<2` alors que ses resamplers tournent en 2.x. Sans épingle, pip rétrograde tout le venv en numpy 1.26 pendant que le torch de l'image est construit contre numpy 2. D'où le seul `numpy==2.3.*` épinglé dans `requirements-onyxia.txt`.

### Deux fichiers de dépendances, et pourquoi

- `requirements.txt` — **épinglé et complet, torch inclus**. C'est celui du dépôt public : un inconnu doit pouvoir reproduire tes figures (exercice 2.8, checklist ML Code Completeness).
- `requirements-onyxia.txt` — **sans torch, sans torchvision**. C'est celui qu'installe le bootstrap, parce que l'image GPU du Datalab fournit déjà un PyTorch appairé à sa version de CUDA, et que le réinstaller casse le GPU.

Écris en commentaire dans les deux fichiers pourquoi ils diffèrent. C'est exactement le genre de détail qui va dans la section « choix d'implémentation et leurs conséquences » du rapport.

---

## 5. Vérification en 60 secondes

```bash
which python                                  # -> .../work/.venvs/ddpm/bin/python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mc ls s3/gfogato/ddpm                         # jeton S3 vivant ? (jamais "mc ls s3/" seul)
git -C ~/work/diffusion-models status -sb     # branche suivie, arbre propre ?
cd ~/work/diffusion-models && pytest -q       # le filet est-il tendu ?
```

Si les cinq passent, tu peux coder. Si `pytest` échoue à froid, corrige **ça** avant toute chose : un dépôt qui ne passe pas ses tests au démarrage est un dépôt qu'un relecteur ne clonera pas deux fois.

---

## 6. Pannes fréquentes

| Symptôme | Cause | Geste |
|---|---|---|
| `mc` ou `aws s3` reste bloqué sans rien afficher | commande lancée sans bucket (`mc ls s3/`) : `ListBuckets` est refusé par la politique `stsonly` | viser `s3/gfogato/...` explicitement |
| Service en **rouge** dans « Mes services », `mc` refuse | jeton MinIO expiré (7 j) | ouvrir un nouveau service, ou copier les scripts de renouvellement depuis <https://datalab.sspcloud.fr/account/storage> |
| `git push` demande un mot de passe | dépôt cloné sans jeton dans l'URL | `git remote set-url origin https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/fogatogl/diffusion-models.git` |
| Commits non rattachés au profil GitHub | e-mail du compte Datalab ≠ e-mail GitHub | corriger dans le compte, puis `git config --global user.email` |
| `python` pointe sur `/opt/conda/bin/python` | `~/.bashrc` réinitialisé | `source ~/.bashrc`, ou relancer le bootstrap |
| Kernel « Python (ddpm) » absent de VS Code | `~/.local` non persistant | relancer le bootstrap, puis recharger la fenêtre VS Code |
| `torch.cuda.is_available()` = `False` | service lancé sans GPU | vérifier l'onglet Ressources ; sans GPU pour un module CPU, c'est normal et voulu |
| `pull` refusé, « travail local non commité » | modifications non poussées à la session précédente | voir §7 : c'est une faute de fin de session, pas de démarrage |
| Tout a disparu | persistance non cochée au lancement | tout rejouer depuis Git + S3 ; c'est précisément pour ça que rien ne vit ailleurs |

---

## 7. Fin de session — la checklist qui compte

Une instance Onyxia peut mourir sans prévenir. **Tant que ce n'est pas poussé, ça n'existe pas.**

```bash
cd ~/work/diffusion-models
git add -A && git commit -m "..." && git push        # le code
bash scripts/sync_s3.sh push                         # poids, samples, résultats lourds
```

Puis : une ligne dans `LEARNING.md` si un bug t'a pris plus de 20 minutes, et seulement ensuite supprimer le service.

Ne finis jamais une session sur un « je pousserai demain ». Le dépôt est la pièce centrale des dossiers de stage, et la règle du projet est **un commit visible par semaine**.

---

## 8. Où va quoi

| Objet | Destination | Jamais |
|---|---|---|
| Code, tests, configs, scripts, figures régénérables | **Git** | — |
| Checkpoints, dataset, samples, runs W&B lourds | **S3** (`~/work/ddpm/...`) | dans Git |
| `results/*.json` (petits, requis pour régénérer les figures) | **Git** | — |
| Jeton GitHub, clés | **compte Datalab / Vault** | dans Git, ni dans un notebook |
| venv | `~/work/.venvs/ddpm` | dans Git |

Un `.gitignore` qui couvre `*.pt`, `*.ckpt`, `data/`, `samples/`, `wandb/`, `.venvs/` est à écrire au premier commit, pas au moment où tu pousses par erreur un checkpoint de 500 Mo.

---

## Sources

- Contrôle de version SSP Cloud : <https://docs.sspcloud.fr/content/version-control.html>
- Stockage MinIO : <https://docs.sspcloud.fr/content/storage.html>
- Configuration des services : <https://docs.sspcloud.fr/content/services-configuration.html>

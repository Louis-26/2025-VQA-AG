# uncomment if it is in linux, and need to convert dos to unix
#sed -i 's/\r$//' git_script/git_push.sh
read -p "Enter the name of your feature branch: " BRANCH_NAME

git add .

git commit -m "update"

git push origin "$BRANCH_NAME"
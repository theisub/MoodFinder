from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from models import AlbumInfo
from nlp_processing import find_simillar

def main_page(request):
    
    return render(request,'mf_page.html')

@csrf_exempt
def album_id(request,pk=1):

    result =find_simillar(request.POST['users_mood'])
    album_info = AlbumInfo.objects.get(pk=pk)
    context = {
        'album_id': album_info.album_id,
        'artist_name': album_info.artist_name,
        'album_name': album_info.album_name,
        'album_cover': album_info.album_cover
    }
    print(context)
    # return HttpResponse(request,'album_index.html',{})
    print('HEY! ur mood is ?', request.POST['users_mood'],'?')
    return render(request,"mf_page.html",{'album_name':result['album_name'],"album_artist":result['artist_name'],"album_cover":result['album_cover'],"sentence":request.POST['users_mood']})